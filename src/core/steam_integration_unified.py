#!/usr/bin/env python3
"""
Unified Steam Platform Integration

This module provides a unified interface to all Steam platform integration
features, making it easy to use the comprehensive Steam Deck integration
system with the Enhanced ML Predictor.

Key Features:
- Single point of access for all Steam integration features
- Automatic component initialization and coordination
- Simplified API for common operations
- Steam Deck hardware optimization
- Gaming Mode seamless operation
- Background service management
- Comprehensive status reporting

Usage:
    from steam_integration_unified import SteamIntegration
    
    # Initialize with automatic configuration
    steam = SteamIntegration()
    await steam.initialize()
    
    # Start background service
    await steam.start_service()
    
    # Get comprehensive status
    status = steam.get_status()
"""

import os
import sys
import time
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# UNIFIED INTEGRATION STATUS
# =============================================================================

class IntegrationStatus(Enum):
    """Overall integration status"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    GAMING_MODE = "gaming_mode"
    ERROR = "error"

@dataclass
class ComponentStatus:
    """Individual component status"""
    name: str
    available: bool
    initialized: bool
    running: bool
    error: Optional[str]
    metrics: Dict[str, Any]

@dataclass
class SteamIntegrationState:
    """Complete Steam integration state"""
    overall_status: IntegrationStatus
    steam_deck_detected: bool
    gaming_mode_active: bool
    steam_running: bool
    
    # Component statuses
    components: Dict[str, ComponentStatus]
    
    # Hardware state
    hardware_state: Optional[Dict[str, Any]]
    
    # Service metrics
    predictions_made: int
    games_optimized: int
    cache_hits: int
    thermal_events: int
    uptime_seconds: float
    
    # Configuration
    configuration: Dict[str, Any]

# =============================================================================
# UNIFIED STEAM INTEGRATION
# =============================================================================

class SteamIntegration:
    """
    Unified Steam platform integration interface
    """
    
    def __init__(self, auto_configure: bool = True):
        self.status = IntegrationStatus.NOT_INITIALIZED
        self.auto_configure = auto_configure
        self.components: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {
            'game_launch': [],
            'game_termination': [],
            'hardware_change': [],
            'gaming_mode_change': [],
            'status_change': []
        }
        
        # Component references
        self.platform_integration = None
        self.dbus_interface = None
        self.hardware_monitor = None
        self.filesystem_manager = None
        self.background_service = None
        self.service_manager = None
        
        logger.info("Unified Steam Integration created")
    
    async def initialize(self) -> bool:
        """Initialize all Steam integration components"""
        if self.status != IntegrationStatus.NOT_INITIALIZED:
            logger.warning(f"Already initialized with status: {self.status.value}")
            return self.status != IntegrationStatus.ERROR
        
        logger.info("Initializing unified Steam integration")
        self.status = IntegrationStatus.INITIALIZING
        
        try:
            # Initialize components in dependency order
            await self._initialize_filesystem_manager()
            await self._initialize_hardware_monitor()
            await self._initialize_dbus_interface()
            await self._initialize_platform_integration()
            await self._initialize_service_manager()
            
            # Setup inter-component communication
            self._setup_component_communication()
            
            self.status = IntegrationStatus.READY
            logger.info("Unified Steam integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Steam integration: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    async def _initialize_filesystem_manager(self) -> None:
        """Initialize filesystem manager"""
        try:
            from .steamos_filesystem_manager import get_filesystem_manager
            self.filesystem_manager = get_filesystem_manager()
            
            # Ensure directory structure
            self.filesystem_manager.ensure_directory_structure()
            
            self.components['filesystem_manager'] = ComponentStatus(
                name='filesystem_manager',
                available=True,
                initialized=True,
                running=True,
                error=None,
                metrics={}
            )
            
            logger.info("Filesystem manager initialized")
            
        except Exception as e:
            logger.error(f"Filesystem manager initialization failed: {e}")
            self.components['filesystem_manager'] = ComponentStatus(
                name='filesystem_manager',
                available=False,
                initialized=False,
                running=False,
                error=str(e),
                metrics={}
            )
    
    async def _initialize_hardware_monitor(self) -> None:
        """Initialize hardware monitor"""
        try:
            from .steam_deck_hardware_integration import get_hardware_monitor
            self.hardware_monitor = get_hardware_monitor()
            
            # Add callback for hardware state changes
            self.hardware_monitor.add_state_callback(self._on_hardware_state_change)
            
            self.components['hardware_monitor'] = ComponentStatus(
                name='hardware_monitor',
                available=True,
                initialized=True,
                running=False,  # Will be True when monitoring starts
                error=None,
                metrics={}
            )
            
            logger.info("Hardware monitor initialized")
            
        except Exception as e:
            logger.error(f"Hardware monitor initialization failed: {e}")
            self.components['hardware_monitor'] = ComponentStatus(
                name='hardware_monitor',
                available=False,
                initialized=False,
                running=False,
                error=str(e),
                metrics={}
            )
    
    async def _initialize_dbus_interface(self) -> None:
        """Initialize D-Bus interface"""
        try:
            from .steam_dbus_interface import get_steam_dbus
            self.dbus_interface = get_steam_dbus()
            
            # Initialize D-Bus interface
            dbus_success = await self.dbus_interface.initialize()
            
            if dbus_success:
                # Add event handlers
                self.dbus_interface.add_event_handler('steam_game', self._on_game_event)
                self.dbus_interface.add_event_handler('gaming_mode', self._on_gaming_mode_event)
                self.dbus_interface.add_event_handler('hardware', self._on_hardware_event)
            
            self.components['dbus_interface'] = ComponentStatus(
                name='dbus_interface',
                available=dbus_success,
                initialized=dbus_success,
                running=False,  # Will be True when monitoring starts
                error=None if dbus_success else "D-Bus initialization failed",
                metrics=self.dbus_interface.get_capabilities() if dbus_success else {}
            )
            
            logger.info(f"D-Bus interface initialized: {dbus_success}")
            
        except Exception as e:
            logger.error(f"D-Bus interface initialization failed: {e}")
            self.components['dbus_interface'] = ComponentStatus(
                name='dbus_interface',
                available=False,
                initialized=False,
                running=False,
                error=str(e),
                metrics={}
            )
    
    async def _initialize_platform_integration(self) -> None:
        """Initialize platform integration"""
        try:
            from .steam_platform_integration import get_steam_integration
            self.platform_integration = get_steam_integration()
            
            # Initialize platform integration
            await self.platform_integration.initialize()
            
            self.components['platform_integration'] = ComponentStatus(
                name='platform_integration',
                available=True,
                initialized=True,
                running=False,  # Will be True when background service starts
                error=None,
                metrics={}
            )
            
            logger.info("Platform integration initialized")
            
        except Exception as e:
            logger.error(f"Platform integration initialization failed: {e}")
            self.components['platform_integration'] = ComponentStatus(
                name='platform_integration',
                available=False,
                initialized=False,
                running=False,
                error=str(e),
                metrics={}
            )
    
    async def _initialize_service_manager(self) -> None:
        """Initialize service manager"""
        try:
            from .steam_background_service import get_service_manager
            self.service_manager = get_service_manager()
            
            self.components['service_manager'] = ComponentStatus(
                name='service_manager',
                available=True,
                initialized=True,
                running=False,  # Will be True when service starts
                error=None,
                metrics={}
            )
            
            logger.info("Service manager initialized")
            
        except Exception as e:
            logger.error(f"Service manager initialization failed: {e}")
            self.components['service_manager'] = ComponentStatus(
                name='service_manager',
                available=False,
                initialized=False,
                running=False,
                error=str(e),
                metrics={}
            )
    
    def _setup_component_communication(self) -> None:
        """Setup communication between components"""
        try:
            # Components are already designed to work together
            # This method can be used for additional coordination
            logger.info("Component communication setup complete")
            
        except Exception as e:
            logger.error(f"Component communication setup failed: {e}")
    
    async def start_service(self) -> bool:
        """Start the background service"""
        if self.status != IntegrationStatus.READY:
            logger.error(f"Cannot start service, status: {self.status.value}")
            return False
        
        logger.info("Starting Steam integration background service")
        
        try:
            # Start D-Bus monitoring
            if self.dbus_interface and self.components['dbus_interface'].available:
                await self.dbus_interface.start_monitoring()
                self.components['dbus_interface'].running = True
            
            # Start hardware monitoring
            if self.hardware_monitor and self.components['hardware_monitor'].available:
                await self.hardware_monitor.start_monitoring()
                self.components['hardware_monitor'].running = True
            
            # Start platform integration background service
            if self.platform_integration and self.components['platform_integration'].available:
                await self.platform_integration.start_background_service()
                self.components['platform_integration'].running = True
            
            # Start service manager
            if self.service_manager and self.components['service_manager'].available:
                service_started = await self.service_manager.start_service()
                self.components['service_manager'].running = service_started
            
            self.status = IntegrationStatus.RUNNING
            self._notify_status_change()
            
            logger.info("Steam integration background service started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start background service: {e}")
            self.status = IntegrationStatus.ERROR
            self._notify_status_change()
            return False
    
    async def stop_service(self) -> None:
        """Stop the background service"""
        logger.info("Stopping Steam integration background service")
        
        try:
            # Stop service manager
            if self.service_manager:
                await self.service_manager.stop_service()
                self.components['service_manager'].running = False
            
            # Stop platform integration
            if self.platform_integration:
                self.platform_integration.background_service_active = False
                self.components['platform_integration'].running = False
            
            # Stop hardware monitoring
            if self.hardware_monitor:
                self.hardware_monitor.stop_monitoring()
                self.components['hardware_monitor'].running = False
            
            # Stop D-Bus monitoring
            if self.dbus_interface:
                await self.dbus_interface.stop_monitoring()
                self.components['dbus_interface'].running = False
            
            self.status = IntegrationStatus.READY
            self._notify_status_change()
            
            logger.info("Steam integration background service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping background service: {e}")
    
    def _on_game_event(self, event) -> None:
        """Handle game events"""
        try:
            logger.info(f"Game event: {event.event_type} - App ID: {event.app_id}")
            
            # Notify callbacks
            if event.event_type == 'launched':
                for callback in self.callbacks['game_launch']:
                    try:
                        callback(event.app_id, event)
                    except Exception as e:
                        logger.error(f"Game launch callback error: {e}")
            elif event.event_type == 'terminated':
                for callback in self.callbacks['game_termination']:
                    try:
                        callback(event.app_id, event)
                    except Exception as e:
                        logger.error(f"Game termination callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Game event handler error: {e}")
    
    def _on_gaming_mode_event(self, event) -> None:
        """Handle Gaming Mode events"""
        try:
            logger.info(f"Gaming Mode event: {event.event_type}")
            
            # Update status if Gaming Mode state changed
            if event.event_type in ['activated', 'deactivated']:
                if event.event_type == 'activated':
                    if self.status == IntegrationStatus.RUNNING:
                        self.status = IntegrationStatus.GAMING_MODE
                else:
                    if self.status == IntegrationStatus.GAMING_MODE:
                        self.status = IntegrationStatus.RUNNING
                
                self._notify_status_change()
            
            # Notify callbacks
            for callback in self.callbacks['gaming_mode_change']:
                try:
                    callback(event.event_type, event)
                except Exception as e:
                    logger.error(f"Gaming mode callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Gaming mode event handler error: {e}")
    
    def _on_hardware_event(self, event) -> None:
        """Handle hardware events"""
        try:
            logger.info(f"Hardware event: {event.event_type}")
            
            # Notify callbacks
            for callback in self.callbacks['hardware_change']:
                try:
                    callback(event.event_type, event)
                except Exception as e:
                    logger.error(f"Hardware callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Hardware event handler error: {e}")
    
    def _on_hardware_state_change(self, new_state) -> None:
        """Handle hardware state changes"""
        try:
            logger.debug(f"Hardware state change: {new_state.thermal_state.value}")
            
            # Notify callbacks
            for callback in self.callbacks['hardware_change']:
                try:
                    callback('state_change', new_state)
                except Exception as e:
                    logger.error(f"Hardware state callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Hardware state change handler error: {e}")
    
    def _notify_status_change(self) -> None:
        """Notify status change callbacks"""
        for callback in self.callbacks['status_change']:
            try:
                callback(self.status, self.get_state())
            except Exception as e:
                logger.error(f"Status change callback error: {e}")
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add event callback"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.debug(f"Added {event_type} callback")
        else:
            logger.warning(f"Unknown callback type: {event_type}")
    
    def remove_callback(self, event_type: str, callback: Callable) -> None:
        """Remove event callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            logger.debug(f"Removed {event_type} callback")
    
    def get_state(self) -> SteamIntegrationState:
        """Get comprehensive integration state"""
        # Get hardware state
        hardware_state = None
        if self.hardware_monitor and self.hardware_monitor.hardware_state:
            state = self.hardware_monitor.hardware_state
            hardware_state = {
                'model': state.model.value,
                'power_state': state.power_state.value,
                'thermal_state': state.thermal_state.value,
                'dock_state': state.dock_state.value,
                'cpu_temperature': state.cpu_temperature,
                'battery_percentage': state.battery_percentage,
                'gaming_mode_active': state.gaming_mode_active
            }
        
        # Get service metrics
        predictions_made = 0
        games_optimized = 0
        cache_hits = 0
        thermal_events = 0
        uptime_seconds = 0
        
        if self.service_manager:
            service_status = self.service_manager.get_service_status()
            metrics = service_status.get('metrics', {})
            predictions_made = metrics.get('predictions_made', 0)
            games_optimized = metrics.get('games_optimized', 0)
            cache_hits = metrics.get('cache_hits', 0)
            thermal_events = metrics.get('thermal_throttle_events', 0)
            uptime_seconds = metrics.get('uptime_seconds', 0)
        
        # Get system states
        steam_deck_detected = False
        gaming_mode_active = False
        steam_running = False
        
        if self.hardware_monitor:
            steam_deck_detected = self.hardware_monitor.is_steam_deck
            if self.hardware_monitor.hardware_state:
                gaming_mode_active = self.hardware_monitor.hardware_state.gaming_mode_active
        
        if self.platform_integration:
            platform_status = self.platform_integration.get_platform_status()
            steam_running = platform_status.steam_running
            gaming_mode_active = gaming_mode_active or platform_status.gaming_mode_active
        
        # Configuration
        configuration = {}
        if self.service_manager:
            config = self.service_manager.load_config()
            configuration = {
                'integration_mode': config.integration_mode.value,
                'enable_dbus_monitoring': config.enable_dbus_monitoring,
                'enable_hardware_monitoring': config.enable_hardware_monitoring,
                'gaming_mode_detection': config.gaming_mode_detection,
                'prediction_frequency_hz': config.prediction_frequency_hz
            }
        
        return SteamIntegrationState(
            overall_status=self.status,
            steam_deck_detected=steam_deck_detected,
            gaming_mode_active=gaming_mode_active,
            steam_running=steam_running,
            components=self.components.copy(),
            hardware_state=hardware_state,
            predictions_made=predictions_made,
            games_optimized=games_optimized,
            cache_hits=cache_hits,
            thermal_events=thermal_events,
            uptime_seconds=uptime_seconds,
            configuration=configuration
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get status as dictionary"""
        state = self.get_state()
        
        return {
            'overall_status': state.overall_status.value,
            'steam_deck_detected': state.steam_deck_detected,
            'gaming_mode_active': state.gaming_mode_active,
            'steam_running': state.steam_running,
            'components': {
                name: {
                    'available': comp.available,
                    'initialized': comp.initialized,
                    'running': comp.running,
                    'error': comp.error
                }
                for name, comp in state.components.items()
            },
            'hardware_state': state.hardware_state,
            'metrics': {
                'predictions_made': state.predictions_made,
                'games_optimized': state.games_optimized,
                'cache_hits': state.cache_hits,
                'thermal_events': state.thermal_events,
                'uptime_seconds': state.uptime_seconds
            },
            'configuration': state.configuration
        }
    
    async def predict_for_game(self, app_id: str) -> Dict[str, Any]:
        """Trigger shader prediction for a specific game"""
        if self.platform_integration:
            try:
                await self.platform_integration._handle_game_launch(app_id)
                return {'success': True, 'app_id': app_id}
            except Exception as e:
                logger.error(f"Game prediction error for {app_id}: {e}")
                return {'success': False, 'app_id': app_id, 'error': str(e)}
        else:
            return {'success': False, 'app_id': app_id, 'error': 'Platform integration not available'}
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get Steam Deck hardware capabilities"""
        if self.hardware_monitor:
            return self.hardware_monitor.get_hardware_capabilities()
        else:
            return {'is_steam_deck': False, 'error': 'Hardware monitor not available'}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        recommendations = []
        
        try:
            # Get hardware-based recommendations
            if self.hardware_monitor and self.hardware_monitor.hardware_state:
                state = self.hardware_monitor.hardware_state
                
                if state.thermal_state.value == 'critical':
                    recommendations.append({
                        'type': 'thermal',
                        'priority': 'high',
                        'title': 'Critical Thermal State',
                        'description': f'CPU temperature is {state.cpu_temperature:.1f}°C',
                        'action': 'Consider reducing shader prediction frequency or improving cooling'
                    })
                
                if state.power_state.value == 'battery_critical':
                    recommendations.append({
                        'type': 'power',
                        'priority': 'high',
                        'title': 'Critical Battery Level',
                        'description': f'Battery at {state.battery_percentage:.0f}%',
                        'action': 'Switch to power saving mode or connect charger'
                    })
                
                if state.gaming_mode_active:
                    recommendations.append({
                        'type': 'gaming',
                        'priority': 'info',
                        'title': 'Gaming Mode Active',
                        'description': 'Background optimization is active',
                        'action': 'Shader prediction is running with minimal impact'
                    })
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
        
        return recommendations

# =============================================================================
# CONTEXT MANAGERS AND UTILITIES
# =============================================================================

@asynccontextmanager
async def steam_integration_context(auto_start_service: bool = True):
    """Context manager for Steam integration"""
    integration = SteamIntegration()
    
    try:
        # Initialize
        success = await integration.initialize()
        if not success:
            raise RuntimeError("Failed to initialize Steam integration")
        
        # Start service if requested
        if auto_start_service:
            await integration.start_service()
        
        yield integration
        
    finally:
        # Cleanup
        if integration.status in [IntegrationStatus.RUNNING, IntegrationStatus.GAMING_MODE]:
            await integration.stop_service()

# Global integration instance
_steam_integration: Optional[SteamIntegration] = None

def get_unified_steam_integration() -> SteamIntegration:
    """Get or create the global unified Steam integration"""
    global _steam_integration
    if _steam_integration is None:
        _steam_integration = SteamIntegration()
    return _steam_integration

# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

async def demonstrate_integration():
    """Demonstrate the unified Steam integration"""
    print("\n🎮 Unified Steam Integration Demonstration")
    print("=" * 55)
    
    # Create integration
    integration = SteamIntegration()
    
    # Initialize
    print(f"\n🚀 Initializing Steam integration...")
    success = await integration.initialize()
    print(f"  Initialization: {'✅' if success else '❌'}")
    
    if success:
        # Show status
        status = integration.get_status()
        print(f"\n📊 Integration Status:")
        print(f"  Overall status: {status['overall_status']}")
        print(f"  Steam Deck detected: {status['steam_deck_detected']}")
        print(f"  Gaming Mode active: {status['gaming_mode_active']}")
        print(f"  Steam running: {status['steam_running']}")
        
        print(f"\n🔧 Component Status:")
        for name, comp in status['components'].items():
            status_icon = "✅" if comp['available'] and comp['initialized'] else "❌"
            running_icon = "🟢" if comp['running'] else "🔴"
            print(f"  {status_icon} {running_icon} {name}")
            if comp['error']:
                print(f"    Error: {comp['error']}")
        
        # Show hardware capabilities
        hardware_caps = integration.get_hardware_capabilities()
        if hardware_caps.get('is_steam_deck'):
            print(f"\n⚙️ Hardware Capabilities:")
            print(f"  Model: {hardware_caps.get('model', 'unknown')}")
            print(f"  OLED display: {hardware_caps.get('has_oled_display', False)}")
            print(f"  APU architecture: {hardware_caps.get('apu_architecture', 'unknown')}")
            print(f"  Memory: {hardware_caps.get('memory_gb', 0)}GB {hardware_caps.get('memory_type', '')}")
        
        # Show optimization recommendations
        recommendations = integration.get_optimization_recommendations()
        if recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for rec in recommendations:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "ℹ️"}.get(rec['priority'], "ℹ️")
                print(f"  {priority_icon} {rec['title']}: {rec['description']}")
        else:
            print(f"\n✅ No optimization recommendations - system running optimally")
        
        # Start service
        print(f"\n🔄 Starting background service...")
        service_started = await integration.start_service()
        print(f"  Service started: {'✅' if service_started else '❌'}")
        
        if service_started:
            # Run for a few seconds
            print(f"\n👀 Running service for 10 seconds...")
            
            def status_callback(new_status, state):
                print(f"  Status change: {new_status.value}")
            
            integration.add_callback('status_change', status_callback)
            
            await asyncio.sleep(10)
            
            # Show final status
            final_status = integration.get_status()
            print(f"\n📈 Final Metrics:")
            metrics = final_status['metrics']
            print(f"  Uptime: {metrics['uptime_seconds']:.1f}s")
            print(f"  Predictions made: {metrics['predictions_made']}")
            print(f"  Games optimized: {metrics['games_optimized']}")
            
            # Stop service
            print(f"\n🛑 Stopping service...")
            await integration.stop_service()
    
    print(f"\n✅ Steam integration demonstration completed")

async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        await demonstrate_integration()
    else:
        print("Unified Steam Platform Integration")
        print("Usage: --demo to run demonstration")

if __name__ == "__main__":
    asyncio.run(main())