#!/usr/bin/env python3
"""
Shader Prediction Compiler - Main Application
Steam Deck optimized service with ML prediction and P2P distribution
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import asyncio
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import psutil

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "security"))

# Import core modules
from shader_prediction_system import SteamDeckShaderPredictor
from steam_deck_integration import (
    SteamDeckHardwareMonitor,
    SteamDeckGameIntegration,
    ThermalManager
)
from p2p_shader_distribution import P2PShaderDistributionSystem
from security.security_integration import UnifiedSecuritySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Service configuration"""
    mode: str = "auto"  # auto, performance, balanced, powersave
    ml_enabled: bool = True
    p2p_enabled: bool = True
    security_enabled: bool = True
    thermal_monitoring: bool = True
    battery_aware: bool = True
    max_cpu_percent: int = 10
    max_memory_mb: int = 500
    cache_size_gb: float = 2.0
    gamemode_throttle: bool = True
    auto_start: bool = False
    log_level: str = "info"
    
    @classmethod
    def from_file(cls, path: Path) -> 'ServiceConfig':
        """Load configuration from JSON file"""
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()
    
    def save(self, path: Path):
        """Save configuration to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class ShaderPredictService:
    """Main service orchestrator for Shader Prediction Compiler"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.running = False
        self.paused = False
        
        # Initialize components
        self.hardware_monitor = None
        self.thermal_manager = None
        self.ml_predictor = None
        self.p2p_system = None
        self.security_system = None
        self.game_integration = None
        
        # Performance metrics
        self.metrics = {
            'shaders_compiled': 0,
            'shaders_predicted': 0,
            'cache_hits': 0,
            'p2p_shares': 0,
            'thermal_throttles': 0,
            'uptime': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGUSR1, self._handle_health_check)
        signal.signal(signal.SIGUSR2, self._handle_pause)
        signal.signal(signal.SIGCONT, self._handle_resume)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def _handle_health_check(self, signum, frame):
        """Handle health check signal"""
        logger.debug("Health check requested")
        # Respond to health check by updating status
        if self.running and not self.paused:
            logger.info("Health check: HEALTHY")
        else:
            logger.warning(f"Health check: DEGRADED (running={self.running}, paused={self.paused})")
    
    def _handle_pause(self, signum, frame):
        """Handle pause signal (for gaming mode)"""
        logger.info("Pausing service for gaming mode")
        self.paused = True
        self._apply_pause()
    
    def _handle_resume(self, signum, frame):
        """Handle resume signal"""
        logger.info("Resuming service")
        self.paused = False
        self._apply_resume()
    
    def initialize(self):
        """Initialize all service components"""
        logger.info("Initializing Shader Prediction Service...")
        
        try:
            # Hardware monitoring
            if self.config.thermal_monitoring:
                logger.info("Initializing hardware monitor...")
                self.hardware_monitor = SteamDeckHardwareMonitor()
                self.hardware_monitor.start_monitoring()
                
                self.thermal_manager = ThermalManager(
                    self.hardware_monitor,
                    throttle_callback=self._handle_thermal_throttle
                )
            
            # ML Prediction System
            if self.config.ml_enabled:
                logger.info("Initializing ML predictor...")
                self.ml_predictor = SteamDeckShaderPredictor(
                    model_path=Path.home() / ".cache/shader-predict-compile/ml_models",
                    enable_neural_net=torch_available(),
                    power_mode=self.config.mode
                )
                self.ml_predictor.load_or_initialize_models()
            
            # P2P Distribution System
            if self.config.p2p_enabled:
                logger.info("Initializing P2P system...")
                self.p2p_system = P2PShaderDistributionSystem(
                    node_id=generate_node_id(),
                    port=8765,
                    cache_dir=Path.home() / ".cache/shader-predict-compile/p2p_cache"
                )
            
            # Security System
            if self.config.security_enabled:
                logger.info("Initializing security system...")
                self.security_system = UnifiedSecuritySystem(
                    config_dir=Path.home() / ".config/shader-predict-compile/security"
                )
                self.security_system.initialize()
            
            # Game Integration
            logger.info("Initializing game integration...")
            self.game_integration = SteamDeckGameIntegration(
                steam_dir=Path.home() / ".local/share/Steam",
                ml_predictor=self.ml_predictor,
                p2p_system=self.p2p_system
            )
            
            logger.info("Service initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    def _handle_thermal_throttle(self, level: str):
        """Handle thermal throttling events"""
        self.metrics['thermal_throttles'] += 1
        
        if level == "critical":
            logger.warning("Critical thermal state - pausing service")
            self.paused = True
            self._apply_pause()
        elif level == "high":
            logger.info("High thermal state - reducing activity")
            self._reduce_activity(0.3)
        elif level == "moderate":
            logger.info("Moderate thermal state - limiting activity")
            self._reduce_activity(0.6)
        else:
            logger.info("Normal thermal state - resuming full activity")
            self._reduce_activity(1.0)
    
    def _reduce_activity(self, factor: float):
        """Reduce service activity by factor (0.0 to 1.0)"""
        if self.ml_predictor:
            self.ml_predictor.set_activity_level(factor)
        if self.p2p_system:
            self.p2p_system.set_bandwidth_limit(int(1024 * 1024 * factor))  # MB/s
    
    def _apply_pause(self):
        """Apply pause to all components"""
        if self.ml_predictor:
            self.ml_predictor.pause()
        if self.p2p_system:
            self.p2p_system.pause()
    
    def _apply_resume(self):
        """Resume all components"""
        if self.ml_predictor:
            self.ml_predictor.resume()
        if self.p2p_system:
            self.p2p_system.resume()
    
    async def run_async(self):
        """Main async service loop"""
        logger.info("Starting async service loop...")
        
        # Start P2P system if enabled
        if self.p2p_system:
            await self.p2p_system.start()
        
        # Main service loop
        while self.running:
            try:
                if not self.paused:
                    # Check for new games
                    new_games = self.game_integration.detect_new_games()
                    for game_id in new_games:
                        logger.info(f"New game detected: {game_id}")
                        await self._process_game(game_id)
                    
                    # Process shader compilation queue
                    await self._process_shader_queue()
                    
                    # Update metrics
                    self.metrics['uptime'] = time.time() - self.start_time
                    
                    # Log status periodically
                    if int(time.time()) % 60 == 0:
                        self._log_status()
                
                # Sleep with interrupt handling
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in service loop: {e}")
                await asyncio.sleep(10)
    
    async def _process_game(self, game_id: str):
        """Process shaders for a specific game"""
        logger.info(f"Processing shaders for game {game_id}")
        
        try:
            # Get game shader patterns
            if self.ml_predictor:
                predictions = await self.ml_predictor.predict_game_shaders(game_id)
                logger.info(f"Predicted {len(predictions)} shaders for game {game_id}")
                
                # Validate with security system
                if self.security_system:
                    validated = []
                    for shader in predictions:
                        if await self.security_system.validate_shader(shader):
                            validated.append(shader)
                    predictions = validated
                
                # Share via P2P if available
                if self.p2p_system and predictions:
                    await self.p2p_system.share_shaders(game_id, predictions)
                    self.metrics['p2p_shares'] += len(predictions)
                
                self.metrics['shaders_predicted'] += len(predictions)
                
        except Exception as e:
            logger.error(f"Failed to process game {game_id}: {e}")
    
    async def _process_shader_queue(self):
        """Process pending shader compilations"""
        # This would integrate with the actual shader compilation system
        pass
    
    def _log_status(self):
        """Log current service status"""
        logger.info(f"Service Status: {self.metrics}")
        
        # Log resource usage
        process = psutil.Process()
        logger.info(f"Resource Usage: CPU={process.cpu_percent()}%, "
                   f"Memory={process.memory_info().rss / 1024 / 1024:.1f}MB")
        
        # Log thermal state
        if self.hardware_monitor:
            state = self.hardware_monitor.get_current_state()
            if state:
                logger.info(f"Thermal: CPU={state.cpu_temp}°C, GPU={state.gpu_temp}°C")
    
    def run(self):
        """Run the service"""
        self.running = True
        self.start_time = time.time()
        
        logger.info("Shader Prediction Service starting...")
        
        # Initialize components
        self.initialize()
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async service
            loop.run_until_complete(self.run_async())
        except KeyboardInterrupt:
            logger.info("Service interrupted")
        finally:
            # Cleanup
            loop.run_until_complete(self.cleanup_async())
            loop.close()
    
    async def cleanup_async(self):
        """Async cleanup"""
        if self.p2p_system:
            await self.p2p_system.stop()
    
    def shutdown(self):
        """Shutdown the service"""
        logger.info("Shutting down service...")
        self.running = False
        
        # Stop components
        if self.hardware_monitor:
            self.hardware_monitor.stop_monitoring()
        
        if self.ml_predictor:
            self.ml_predictor.save_models()
        
        if self.security_system:
            self.security_system.shutdown()
        
        logger.info("Service shutdown complete")


def torch_available() -> bool:
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False


def generate_node_id() -> str:
    """Generate unique node ID for P2P"""
    import uuid
    import hashlib
    
    # Use hardware info for consistent ID
    hw_info = f"{os.uname().nodename}-{os.getuid()}"
    return hashlib.sha256(hw_info.encode()).hexdigest()[:16]


class SteamDeckGameIntegration:
    """Stub for game integration"""
    def __init__(self, steam_dir, ml_predictor, p2p_system):
        self.steam_dir = steam_dir
        self.ml_predictor = ml_predictor
        self.p2p_system = p2p_system
        self.known_games = set()
    
    def detect_new_games(self):
        """Detect newly installed games"""
        # This would scan Steam library for new games
        return []


class ThermalManager:
    """Stub for thermal management"""
    def __init__(self, hardware_monitor, throttle_callback):
        self.hardware_monitor = hardware_monitor
        self.throttle_callback = throttle_callback


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Shader Prediction Compiler Service')
    parser.add_argument('--config', type=Path, 
                       default=Path.home() / '.config/shader-predict-compile/settings.json',
                       help='Configuration file path')
    parser.add_argument('--cache-dir', type=Path,
                       default=Path.home() / '.cache/shader-predict-compile',
                       help='Cache directory path')
    parser.add_argument('--log-file', type=Path,
                       help='Log file path')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.log_file:
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(handler)
    
    # Load configuration
    config = ServiceConfig.from_file(args.config)
    
    # Create and run service
    service = ShaderPredictService(config)
    
    try:
        service.run()
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()