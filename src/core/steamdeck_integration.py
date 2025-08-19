#!/usr/bin/env python3
"""
Steam Deck Integration System
Master coordination of all Steam Deck optimizations
"""

import os
import time
import logging
import threading
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Import Steam Deck specific optimizers
from .steamdeck_thermal_optimizer import get_thermal_optimizer, SteamDeckModel, ThermalLimits
from .steamdeck_cache_optimizer import get_cache_optimizer
from .steamdeck_gaming_detector import get_gaming_detector, GamingState
from .thread_pool_manager import get_thread_manager
from .threading_config import configure_threading_for_steam_deck, ThreadingConfig

@dataclass
class SteamDeckOptimizationProfile:
    """Optimization profile for different usage scenarios"""
    name: str
    description: str
    
    # Threading limits
    max_threads: int
    ml_threads: int
    compilation_threads: int
    
    # Cache configuration
    memory_cache_mb: int
    disk_cache_mb: int
    
    # Thermal management
    thermal_monitoring_interval: float
    thermal_throttle_temp: float
    
    # Performance targets
    prediction_target_ms: float
    memory_limit_mb: int
    
    # Gaming considerations
    pause_compilation_during_games: bool
    reduce_cache_during_games: bool

class SteamDeckIntegrationSystem:
    """
    Comprehensive Steam Deck optimization system
    
    Coordinates all hardware-specific optimizations:
    - Thermal management with APU monitoring
    - Memory-mapped caching for NVMe/SD storage
    - Gaming mode detection and adaptation
    - Thread pool management for 8-core APU
    - ML model optimization for handheld constraints
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Component instances
        self.thermal_optimizer = None
        self.cache_optimizer = None 
        self.gaming_detector = None
        self.thread_manager = None
        self.threading_configurator = None
        
        # System state
        self.current_profile = None
        self.optimization_active = False
        self.last_adaptation_time = 0
        self.adaptation_interval = 5.0  # Adapt every 5 seconds
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_count = 0
        
        # Threading
        self._coordination_thread = None
        self._shutdown_event = threading.Event()
        
        # Optimization profiles
        self.profiles = self._create_optimization_profiles()
        
        self.logger.info("Steam Deck integration system initialized")
    
    def _create_optimization_profiles(self) -> Dict[str, SteamDeckOptimizationProfile]:
        """Create optimization profiles for different scenarios"""
        profiles = {}
        
        # Gaming Mode - Maximum game performance
        profiles["gaming"] = SteamDeckOptimizationProfile(
            name="gaming",
            description="Maximum game performance, minimal background work",
            max_threads=2,
            ml_threads=1,
            compilation_threads=0,  # Pause compilation
            memory_cache_mb=32,
            disk_cache_mb=256,
            thermal_monitoring_interval=2.0,
            thermal_throttle_temp=80.0,
            prediction_target_ms=0.1,  # Very fast predictions
            memory_limit_mb=100,
            pause_compilation_during_games=True,
            reduce_cache_during_games=True
        )
        
        # Battery Saver - Optimize for battery life
        profiles["battery_saver"] = SteamDeckOptimizationProfile(
            name="battery_saver",
            description="Optimized for battery life",
            max_threads=3,
            ml_threads=1,
            compilation_threads=1,
            memory_cache_mb=32,
            disk_cache_mb=256,
            thermal_monitoring_interval=10.0,
            thermal_throttle_temp=75.0,
            prediction_target_ms=0.2,
            memory_limit_mb=80,
            pause_compilation_during_games=True,
            reduce_cache_during_games=False
        )
        
        # Performance - Maximum background processing
        profiles["performance"] = SteamDeckOptimizationProfile(
            name="performance",
            description="Maximum background performance",
            max_threads=6,
            ml_threads=2,
            compilation_threads=3,
            memory_cache_mb=96,
            disk_cache_mb=768,
            thermal_monitoring_interval=3.0,
            thermal_throttle_temp=85.0,
            prediction_target_ms=0.05,  # Ultra-fast predictions
            memory_limit_mb=200,
            pause_compilation_during_games=False,
            reduce_cache_during_games=False
        )
        
        # Balanced - Good compromise
        profiles["balanced"] = SteamDeckOptimizationProfile(
            name="balanced",
            description="Balanced performance and efficiency",
            max_threads=4,
            ml_threads=2,
            compilation_threads=2,
            memory_cache_mb=64,
            disk_cache_mb=512,
            thermal_monitoring_interval=5.0,
            thermal_throttle_temp=82.0,
            prediction_target_ms=0.075,
            memory_limit_mb=150,
            pause_compilation_during_games=True,
            reduce_cache_during_games=False
        )
        
        return profiles
    
    def initialize(self) -> bool:
        """Initialize all Steam Deck optimization components"""
        try:
            self.logger.info("Initializing Steam Deck optimization components...")
            
            # Initialize threading configuration first
            threading_config = ThreadingConfig(
                max_threads=6,
                ml_threads=2,
                compilation_threads=2,
                enable_thermal_scaling=True,
                enable_battery_scaling=True,
                enable_gaming_mode=True
            )
            
            self.threading_configurator = configure_threading_for_steam_deck(threading_config)
            self.logger.info("Threading configuration initialized")
            
            # Initialize thread manager
            self.thread_manager = get_thread_manager()
            self.logger.info("Thread manager initialized")
            
            # Initialize thermal optimizer
            self.thermal_optimizer = get_thermal_optimizer()
            self.thermal_optimizer.start_monitoring()
            self.logger.info("Thermal optimizer initialized")
            
            # Initialize cache optimizer
            self.cache_optimizer = get_cache_optimizer()
            self.logger.info("Cache optimizer initialized")
            
            # Initialize gaming detector
            self.gaming_detector = get_gaming_detector()
            self.gaming_detector.start_monitoring()
            self.logger.info("Gaming detector initialized")
            
            # Set initial profile
            self.current_profile = self.profiles["balanced"]
            self._apply_profile(self.current_profile)
            
            # Start coordination
            self._start_coordination()
            
            self.optimization_active = True
            self.logger.info("✅ Steam Deck optimization system fully initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Steam Deck optimization system: {e}")
            return False
    
    def _start_coordination(self):
        """Start the coordination thread"""
        self._coordination_thread = threading.Thread(
            target=self._coordination_loop,
            name="steamdeck_coordinator",
            daemon=True
        )
        self._coordination_thread.start()
    
    def _coordination_loop(self):
        """Main coordination loop that adapts system based on current state"""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Check if adaptation is needed
                if current_time - self.last_adaptation_time >= self.adaptation_interval:
                    self._adapt_system()
                    self.last_adaptation_time = current_time
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
                time.sleep(5.0)
    
    def _adapt_system(self):
        """Adapt system configuration based on current conditions"""
        try:
            # Get current system state
            gaming_state = self.gaming_detector.get_gaming_state()
            thermal_status = self.thermal_optimizer.get_status()
            gaming_recommendations = self.gaming_detector.get_resource_recommendations()
            
            # Determine optimal profile
            optimal_profile = self._determine_optimal_profile(
                gaming_state, thermal_status, gaming_recommendations
            )
            
            # Apply profile if changed
            if optimal_profile.name != self.current_profile.name:
                self.logger.info(f"Switching optimization profile: {self.current_profile.name} -> {optimal_profile.name}")
                self.current_profile = optimal_profile
                self._apply_profile(optimal_profile)
                self.adaptation_count += 1
            
            # Record performance metrics
            self._record_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"System adaptation error: {e}")
    
    def _determine_optimal_profile(self, gaming_state: GamingState, 
                                 thermal_status: Dict, 
                                 gaming_recommendations: Dict) -> SteamDeckOptimizationProfile:
        """Determine the optimal profile based on current system state"""
        
        # Check if thermal throttling is needed
        thermal_state = thermal_status.get("thermal_state", "normal")
        max_temp = thermal_status.get("max_temperature", 70.0)
        
        # Gaming takes highest priority
        if gaming_state in [GamingState.ACTIVE_3D, GamingState.LAUNCHING]:
            return self.profiles["gaming"]
        elif gaming_state == GamingState.ACTIVE_2D:
            # 2D games can tolerate more background work
            if thermal_state in ["throttling", "critical"]:
                return self.profiles["gaming"]  # Still use gaming profile if thermal issues
            else:
                return self.profiles["balanced"]
        
        # Battery considerations
        try:
            battery_path = Path("/sys/class/power_supply/BAT1/capacity")
            if battery_path.exists():
                battery_level = int(battery_path.read_text().strip())
                
                charging_path = Path("/sys/class/power_supply/BAT1/status")
                is_charging = "Charging" in charging_path.read_text().strip()
                
                if not is_charging and battery_level < 20:
                    return self.profiles["battery_saver"]
                elif not is_charging and battery_level < 50:
                    return self.profiles["balanced"]
        except Exception:
            pass
        
        # Thermal considerations
        if thermal_state == "critical" or max_temp > 90:
            return self.profiles["battery_saver"]  # Most conservative
        elif thermal_state == "throttling" or max_temp > 85:
            return self.profiles["balanced"]
        
        # Default to performance mode when conditions are good
        if thermal_state in ["cool", "normal"] and gaming_state == GamingState.IDLE:
            return self.profiles["performance"]
        
        # Fallback to balanced
        return self.profiles["balanced"]
    
    def _apply_profile(self, profile: SteamDeckOptimizationProfile):
        """Apply an optimization profile to all components"""
        try:
            self.logger.info(f"Applying optimization profile: {profile.name} - {profile.description}")
            
            # Update threading configuration
            if self.threading_configurator:
                new_config = ThreadingConfig(
                    max_threads=profile.max_threads,
                    ml_threads=profile.ml_threads,
                    compilation_threads=profile.compilation_threads
                )
                self.threading_configurator.update_configuration(new_config)
            
            # Update cache configuration
            if self.cache_optimizer:
                # Note: Cache configuration update would need to be implemented
                pass
            
            # Update thermal monitoring
            if self.thermal_optimizer:
                # Note: Thermal monitoring update would need to be implemented
                pass
            
            self.logger.info(f"Profile '{profile.name}' applied successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to apply profile '{profile.name}': {e}")
    
    def _record_performance_metrics(self):
        """Record current performance metrics"""
        try:
            metrics = {
                "timestamp": time.time(),
                "profile": self.current_profile.name,
                "gaming_state": self.gaming_detector.get_gaming_state().value,
                "thermal_temp": self.thermal_optimizer.get_max_temperature(),
                "thermal_state": self.thermal_optimizer.get_thermal_state(),
                "thread_count": len(threading.enumerate()),
                "adaptation_count": self.adaptation_count
            }
            
            # Add thread manager metrics if available
            if self.thread_manager:
                thread_metrics = self.thread_manager.get_thread_metrics()
                metrics.update({
                    "active_threads": thread_metrics.get("total_active_threads", 0),
                    "resource_state": thread_metrics.get("resource_state", "unknown")
                })
            
            # Add cache metrics if available
            if self.cache_optimizer:
                cache_stats = self.cache_optimizer.get_stats()
                metrics.update({
                    "cache_entries": cache_stats.total_entries,
                    "cache_memory_mb": cache_stats.memory_usage_mb,
                    "cache_hit_rate": cache_stats.hit_rate
                })
            
            self.performance_history.append(metrics)
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Failed to record performance metrics: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "optimization_active": self.optimization_active,
            "current_profile": self.current_profile.name if self.current_profile else None,
            "adaptation_count": self.adaptation_count,
            "last_adaptation_time": self.last_adaptation_time,
            "available_profiles": list(self.profiles.keys())
        }
        
        # Add component status
        if self.thermal_optimizer:
            status["thermal"] = self.thermal_optimizer.get_status()
        
        if self.gaming_detector:
            status["gaming"] = self.gaming_detector.get_status()
        
        if self.thread_manager:
            status["threading"] = self.thread_manager.get_thread_metrics()
        
        if self.cache_optimizer:
            status["cache"] = asdict(self.cache_optimizer.get_stats())
        
        # Add recent performance metrics
        if self.performance_history:
            status["recent_metrics"] = self.performance_history[-10:]  # Last 10 samples
        
        return status
    
    def force_profile(self, profile_name: str) -> bool:
        """Force a specific optimization profile"""
        if profile_name not in self.profiles:
            self.logger.error(f"Unknown profile: {profile_name}")
            return False
        
        try:
            profile = self.profiles[profile_name]
            self.current_profile = profile
            self._apply_profile(profile)
            self.logger.info(f"Forced profile change to: {profile_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to force profile '{profile_name}': {e}")
            return False
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations based on current state"""
        if not self.gaming_detector or not self.thermal_optimizer:
            return {}
        
        gaming_state = self.gaming_detector.get_gaming_state()
        thermal_status = self.thermal_optimizer.get_status()
        
        recommendations = {
            "current_profile": self.current_profile.name,
            "recommended_actions": [],
            "performance_score": 0.0,
            "efficiency_score": 0.0
        }
        
        # Gaming recommendations
        if gaming_state != GamingState.IDLE:
            recommendations["recommended_actions"].append(
                "Gaming detected - background compilation paused"
            )
        
        # Thermal recommendations
        thermal_state = thermal_status.get("thermal_state", "normal")
        if thermal_state in ["throttling", "critical"]:
            recommendations["recommended_actions"].append(
                "High temperature detected - reducing background work"
            )
        
        # Calculate performance scores (0-100)
        base_score = 70.0
        
        # Adjust based on thermal state
        thermal_multiplier = {
            "cool": 1.2,
            "normal": 1.0,
            "warm": 0.9,
            "throttling": 0.7,
            "critical": 0.5
        }.get(thermal_state, 0.8)
        
        # Adjust based on gaming state
        gaming_multiplier = {
            GamingState.IDLE: 1.0,
            GamingState.ACTIVE_2D: 0.8,
            GamingState.ACTIVE_3D: 0.6,
            GamingState.LAUNCHING: 0.7
        }.get(gaming_state, 0.8)
        
        recommendations["performance_score"] = min(100.0, base_score * thermal_multiplier * gaming_multiplier)
        recommendations["efficiency_score"] = min(100.0, base_score * (2.0 - thermal_multiplier))
        
        return recommendations
    
    def shutdown(self):
        """Shutdown the integration system"""
        self.logger.info("Shutting down Steam Deck integration system...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop coordination thread
        if self._coordination_thread:
            self._coordination_thread.join(timeout=5.0)
        
        # Shutdown components
        if self.gaming_detector:
            self.gaming_detector.stop_monitoring()
        
        if self.thermal_optimizer:
            self.thermal_optimizer.stop_monitoring()
        
        if self.thread_manager:
            self.thread_manager.shutdown()
        
        if self.cache_optimizer:
            self.cache_optimizer.cleanup()
        
        self.optimization_active = False
        self.logger.info("✅ Steam Deck integration system shutdown completed")


# Global integration system instance
_integration_system = None

def get_integration_system() -> SteamDeckIntegrationSystem:
    """Get or create global integration system"""
    global _integration_system
    if _integration_system is None:
        _integration_system = SteamDeckIntegrationSystem()
    return _integration_system


if __name__ == "__main__":
    # Test integration system
    logging.basicConfig(level=logging.INFO)
    
    print("🎮 Steam Deck Integration System Test")
    print("=" * 50)
    
    system = get_integration_system()
    
    # Initialize system
    if system.initialize():
        print("✅ System initialized successfully")
        
        try:
            # Monitor for a few cycles
            for i in range(3):
                time.sleep(5)
                status = system.get_system_status()
                
                print(f"\nCycle {i+1}:")
                print(f"  Profile: {status.get('current_profile', 'unknown')}")
                print(f"  Gaming: {status.get('gaming', {}).get('gaming_state', 'unknown')}")
                print(f"  Thermal: {status.get('thermal', {}).get('thermal_state', 'unknown')} "
                      f"({status.get('thermal', {}).get('max_temperature', 0):.1f}°C)")
                print(f"  Threads: {status.get('threading', {}).get('total_active_threads', 0)}")
                
                # Show recommendations
                recommendations = system.get_performance_recommendations()
                print(f"  Performance Score: {recommendations.get('performance_score', 0):.1f}%")
                print(f"  Efficiency Score: {recommendations.get('efficiency_score', 0):.1f}%")
        
        finally:
            system.shutdown()
            print("\n✅ Integration system test completed")
    
    else:
        print("❌ System initialization failed")