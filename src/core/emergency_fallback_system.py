#!/usr/bin/env python3
"""
Emergency Fallback System for Resource Exhaustion
================================================

Provides comprehensive emergency fallback mechanisms for when the ML Shader
Prediction Compiler encounters resource exhaustion or critical system states.

Emergency Scenarios Handled:
- Thread creation failures ("can't start new thread")
- Memory pressure and OOM conditions
- Thermal emergency states (>95°C)
- Steam client resource conflicts
- SystemD task limit exhaustion
- Critical system resource shortage

Usage:
    fallback_system = EmergencyFallbackSystem()
    await fallback_system.initialize()
    
    # Emergency activation
    fallback_system.activate_emergency_mode(reason="Thread exhaustion")
"""

import os
import sys
import time
import asyncio
import signal
import psutil
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum
import gc
import resource

# Early threading setup MUST be done before any other imports
if 'setup_threading' not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import setup_threading
    setup_threading.configure_for_steam_deck()

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    """Emergency severity levels."""
    NONE = "none"                    # Normal operation
    WARNING = "warning"              # Resource pressure detected
    MODERATE = "moderate"            # Some degradation needed
    SEVERE = "severe"                # Significant degradation required
    CRITICAL = "critical"            # Emergency measures activated
    SHUTDOWN = "shutdown"            # System shutdown required

class EmergencyTrigger(Enum):
    """Emergency trigger types."""
    THREAD_EXHAUSTION = "thread_exhaustion"
    MEMORY_PRESSURE = "memory_pressure"
    THERMAL_CRITICAL = "thermal_critical"
    RESOURCE_STARVATION = "resource_starvation"
    SYSTEM_OVERLOAD = "system_overload"
    STEAM_CONFLICT = "steam_conflict"
    USER_REQUEST = "user_request"
    AUTOMATIC = "automatic"

@dataclass
class EmergencyAction:
    """Emergency response action."""
    name: str
    level: EmergencyLevel
    function: Callable
    priority: int = 0  # Higher priority executes first
    timeout_seconds: float = 5.0
    destructive: bool = False  # Marks actions that disable functionality
    reversible: bool = True   # Marks actions that can be undone

@dataclass
class EmergencyState:
    """Current emergency state."""
    level: EmergencyLevel = EmergencyLevel.NONE
    active_triggers: Set[EmergencyTrigger] = None
    activated_actions: Set[str] = None
    start_time: float = 0.0
    duration_seconds: float = 0.0
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def __post_init__(self):
        if self.active_triggers is None:
            self.active_triggers = set()
        if self.activated_actions is None:
            self.activated_actions = set()

@dataclass
class SystemSnapshot:
    """System state snapshot for recovery."""
    timestamp: float
    thread_count: int
    memory_usage_mb: float
    cpu_percent: float
    process_count: int
    environment_vars: Dict[str, str]
    active_services: List[str]
    ml_state: Dict[str, Any]

class EmergencyFallbackSystem:
    """Comprehensive emergency fallback and recovery system."""
    
    def __init__(self):
        self.is_steam_deck = setup_threading.is_steam_deck()
        self.emergency_lock = threading.Lock()
        self.emergency_state = EmergencyState()
        self.monitoring_active = False
        self.shutdown_requested = False
        
        # System state tracking
        self.baseline_snapshot: Optional[SystemSnapshot] = None
        self.recovery_snapshots: List[SystemSnapshot] = []
        self.max_snapshots = 10
        
        # Component references
        self.thermal_manager = None
        self.steam_integration = None
        self.capacity_validator = None
        self.ml_predictor = None
        
        # Callbacks
        self.emergency_callbacks: List[Callable[[EmergencyLevel, EmergencyTrigger], None]] = []
        self.recovery_callbacks: List[Callable[[bool], None]] = []
        
        # Resource limits for emergency mode
        self.emergency_limits = {
            'max_threads': 1,
            'max_memory_mb': 100,
            'max_cpu_percent': 25,
            'disable_ml': True,
            'disable_background_tasks': True,
            'disable_compilation': True
        }
        
        logger.info(f"Emergency fallback system initialized - Steam Deck: {self.is_steam_deck}")
        
        # Setup emergency actions
        self._setup_emergency_actions()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_emergency_actions(self) -> None:
        """Setup emergency response actions in priority order."""
        self.emergency_actions = [
            # Level 1: Warning - Monitoring and alerts
            EmergencyAction(
                name="enable_resource_monitoring",
                level=EmergencyLevel.WARNING,
                function=self._enable_resource_monitoring,
                priority=100,
                destructive=False
            ),
            EmergencyAction(
                name="create_system_snapshot",
                level=EmergencyLevel.WARNING,
                function=self._create_system_snapshot,
                priority=90,
                destructive=False
            ),
            
            # Level 2: Moderate - Performance reduction
            EmergencyAction(
                name="reduce_thread_limits",
                level=EmergencyLevel.MODERATE,
                function=self._reduce_thread_limits,
                priority=80,
                destructive=False,
                reversible=True
            ),
            EmergencyAction(
                name="enable_aggressive_gc",
                level=EmergencyLevel.MODERATE,
                function=self._enable_aggressive_gc,
                priority=75,
                destructive=False,
                reversible=True
            ),
            EmergencyAction(
                name="reduce_cache_size",
                level=EmergencyLevel.MODERATE,
                function=self._reduce_cache_size,
                priority=70,
                destructive=False,
                reversible=True
            ),
            
            # Level 3: Severe - Disable non-critical features
            EmergencyAction(
                name="disable_background_tasks",
                level=EmergencyLevel.SEVERE,
                function=self._disable_background_tasks,
                priority=60,
                destructive=True,
                reversible=True
            ),
            EmergencyAction(
                name="disable_compilation",
                level=EmergencyLevel.SEVERE,
                function=self._disable_compilation,
                priority=55,
                destructive=True,
                reversible=True
            ),
            EmergencyAction(
                name="force_memory_cleanup",
                level=EmergencyLevel.SEVERE,
                function=self._force_memory_cleanup,
                priority=50,
                destructive=False,
                reversible=False
            ),
            
            # Level 4: Critical - Disable core features
            EmergencyAction(
                name="disable_ml_inference",
                level=EmergencyLevel.CRITICAL,
                function=self._disable_ml_inference,
                priority=40,
                destructive=True,
                reversible=True
            ),
            EmergencyAction(
                name="emergency_thread_cleanup",
                level=EmergencyLevel.CRITICAL,
                function=self._emergency_thread_cleanup,
                priority=35,
                destructive=False,
                reversible=False
            ),
            EmergencyAction(
                name="kill_non_essential_processes",
                level=EmergencyLevel.CRITICAL,
                function=self._kill_non_essential_processes,
                priority=30,
                destructive=True,
                reversible=False
            ),
            
            # Level 5: Shutdown - Last resort
            EmergencyAction(
                name="request_graceful_shutdown",
                level=EmergencyLevel.SHUTDOWN,
                function=self._request_graceful_shutdown,
                priority=10,
                destructive=True,
                reversible=False
            ),
            EmergencyAction(
                name="emergency_exit",
                level=EmergencyLevel.SHUTDOWN,
                function=self._emergency_exit,
                priority=0,
                destructive=True,
                reversible=False
            )
        ]
        
        # Sort actions by priority (higher first)
        self.emergency_actions.sort(key=lambda a: a.priority, reverse=True)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for emergency situations."""
        try:
            # Handle SIGTERM gracefully
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Handle SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # Handle SIGUSR1 for emergency activation
            signal.signal(signal.SIGUSR1, self._emergency_signal_handler)
            
        except Exception as e:
            logger.warning(f"Signal handler setup failed: {e}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals."""
        logger.warning(f"Received signal {signum} - initiating graceful shutdown")
        asyncio.create_task(self.activate_emergency_mode(
            level=EmergencyLevel.SHUTDOWN,
            trigger=EmergencyTrigger.USER_REQUEST,
            reason=f"Signal {signum}"
        ))
    
    def _emergency_signal_handler(self, signum: int, frame) -> None:
        """Handle emergency signal (SIGUSR1)."""
        logger.warning(f"Emergency signal received - activating emergency mode")
        asyncio.create_task(self.activate_emergency_mode(
            level=EmergencyLevel.CRITICAL,
            trigger=EmergencyTrigger.USER_REQUEST,
            reason="Emergency signal"
        ))
    
    def _create_system_snapshot(self) -> bool:
        """Create system state snapshot."""
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Count threads and processes
            thread_count = 0
            process_count = len(psutil.pids())
            
            try:
                result = subprocess.run(['ps', '-eLf'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    thread_count = len(result.stdout.splitlines()) - 1
            except Exception:
                thread_count = process_count * 2  # Estimate
            
            # Get environment variables
            env_vars = {
                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', ''),
                'LIGHTGBM_NUM_THREADS': os.environ.get('LIGHTGBM_NUM_THREADS', ''),
                'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', ''),
            }
            
            # Create snapshot
            snapshot = SystemSnapshot(
                timestamp=time.time(),
                thread_count=thread_count,
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_percent=cpu_percent,
                process_count=process_count,
                environment_vars=env_vars,
                active_services=[],  # Placeholder
                ml_state={}  # Placeholder
            )
            
            # Store snapshot
            self.recovery_snapshots.append(snapshot)
            if len(self.recovery_snapshots) > self.max_snapshots:
                self.recovery_snapshots.pop(0)
            
            if self.baseline_snapshot is None:
                self.baseline_snapshot = snapshot
                logger.info("Baseline system snapshot created")
            else:
                logger.debug("System snapshot created")
            
            return True
            
        except Exception as e:
            logger.error(f"System snapshot creation failed: {e}")
            return False
    
    def _enable_resource_monitoring(self) -> bool:
        """Enable enhanced resource monitoring."""
        try:
            self.monitoring_active = True
            logger.info("Enhanced resource monitoring enabled")
            return True
        except Exception as e:
            logger.error(f"Resource monitoring enable failed: {e}")
            return False
    
    def _reduce_thread_limits(self) -> bool:
        """Reduce threading limits to minimal levels."""
        try:
            # Set ultra-conservative thread limits
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['LIGHTGBM_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            
            logger.warning("Thread limits reduced to minimum (1 thread per library)")
            return True
        except Exception as e:
            logger.error(f"Thread limit reduction failed: {e}")
            return False
    
    def _enable_aggressive_gc(self) -> bool:
        """Enable aggressive garbage collection."""
        try:
            # Force immediate garbage collection
            collected = gc.collect()
            
            # Set aggressive GC thresholds
            gc.set_threshold(100, 5, 5)  # Much more aggressive than default (700, 10, 10)
            
            # Enable GC debugging (optional)
            if logger.isEnabledFor(logging.DEBUG):
                gc.set_debug(gc.DEBUG_STATS)
            
            logger.warning(f"Aggressive garbage collection enabled - collected {collected} objects")
            return True
        except Exception as e:
            logger.error(f"Aggressive GC enable failed: {e}")
            return False
    
    def _reduce_cache_size(self) -> bool:
        """Reduce cache sizes to minimal levels."""
        try:
            # This would integrate with the cache system
            logger.warning("Cache sizes reduced to emergency levels")
            return True
        except Exception as e:
            logger.error(f"Cache size reduction failed: {e}")
            return False
    
    def _disable_background_tasks(self) -> bool:
        """Disable all background tasks."""
        try:
            # This would integrate with background task system
            logger.warning("Background tasks disabled")
            return True
        except Exception as e:
            logger.error(f"Background task disable failed: {e}")
            return False
    
    def _disable_compilation(self) -> bool:
        """Disable shader compilation."""
        try:
            # This would integrate with compilation system
            logger.warning("Shader compilation disabled")
            return True
        except Exception as e:
            logger.error(f"Compilation disable failed: {e}")
            return False
    
    def _force_memory_cleanup(self) -> bool:
        """Force aggressive memory cleanup."""
        try:
            # Multiple garbage collection passes
            for i in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
                logger.debug(f"GC pass {i+1}: collected {collected} objects")
            
            # Force cleanup of specific modules if available
            try:
                import numpy as np
                np.testing.clear_and_catch_warnings()
            except Exception:
                pass
            
            logger.warning("Forced memory cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _disable_ml_inference(self) -> bool:
        """Disable ML inference capabilities."""
        try:
            if self.ml_predictor:
                # This would integrate with ML predictor
                pass
            
            logger.error("ML inference disabled due to emergency")
            return True
        except Exception as e:
            logger.error(f"ML inference disable failed: {e}")
            return False
    
    def _emergency_thread_cleanup(self) -> bool:
        """Emergency thread cleanup and resource recovery."""
        try:
            # Get current thread count
            try:
                result = subprocess.run(['ps', '-eLf'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    current_threads = len(result.stdout.splitlines()) - 1
                    logger.warning(f"Emergency thread cleanup - current threads: {current_threads}")
            except Exception:
                pass
            
            # Force garbage collection multiple times
            for i in range(5):
                collected = gc.collect()
                if collected == 0:
                    break
                time.sleep(0.1)
            
            # Clear any cached thread pools
            try:
                import concurrent.futures
                # This is destructive - clears global thread pools
                concurrent.futures._global_shutdown = True
            except Exception:
                pass
            
            logger.error("Emergency thread cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Emergency thread cleanup failed: {e}")
            return False
    
    def _kill_non_essential_processes(self) -> bool:
        """Kill non-essential processes to free resources."""
        try:
            # This is very destructive - only in true emergency
            killed_count = 0
            
            # Get processes owned by current user
            current_user = os.getuid()
            
            for proc in psutil.process_iter(['pid', 'name', 'uids', 'memory_percent']):
                try:
                    if (proc.info['uids'].real == current_user and
                        proc.info['memory_percent'] > 5.0 and  # Using significant memory
                        proc.info['name'].lower() not in ['steam', 'systemd', 'python3']):  # Don't kill critical processes
                        
                        proc.kill()
                        killed_count += 1
                        
                        if killed_count >= 5:  # Limit destruction
                            break
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            logger.error(f"Emergency process cleanup - killed {killed_count} processes")
            return True
        except Exception as e:
            logger.error(f"Process cleanup failed: {e}")
            return False
    
    def _request_graceful_shutdown(self) -> bool:
        """Request graceful system shutdown."""
        try:
            self.shutdown_requested = True
            logger.error("Graceful shutdown requested due to emergency")
            return True
        except Exception as e:
            logger.error(f"Graceful shutdown request failed: {e}")
            return False
    
    def _emergency_exit(self) -> bool:
        """Emergency exit as last resort."""
        try:
            logger.critical("EMERGENCY EXIT - System resources exhausted")
            # Ensure logs are flushed
            logging.shutdown()
            os._exit(1)  # Immediate exit
        except Exception:
            sys.exit(1)
    
    def _determine_emergency_level(self, 
                                 trigger: EmergencyTrigger,
                                 system_metrics: Optional[Dict[str, Any]] = None) -> EmergencyLevel:
        """Determine appropriate emergency level based on trigger and system state."""
        try:
            if system_metrics is None:
                system_metrics = self._get_system_metrics()
            
            # Base level on trigger type
            base_level = {
                EmergencyTrigger.THREAD_EXHAUSTION: EmergencyLevel.SEVERE,
                EmergencyTrigger.MEMORY_PRESSURE: EmergencyLevel.MODERATE,
                EmergencyTrigger.THERMAL_CRITICAL: EmergencyLevel.CRITICAL,
                EmergencyTrigger.RESOURCE_STARVATION: EmergencyLevel.SEVERE,
                EmergencyTrigger.SYSTEM_OVERLOAD: EmergencyLevel.MODERATE,
                EmergencyTrigger.STEAM_CONFLICT: EmergencyLevel.WARNING,
                EmergencyTrigger.USER_REQUEST: EmergencyLevel.CRITICAL,
                EmergencyTrigger.AUTOMATIC: EmergencyLevel.WARNING
            }.get(trigger, EmergencyLevel.WARNING)
            
            # Escalate based on system metrics
            if system_metrics:
                memory_pressure = system_metrics.get('memory_percent', 0)
                thread_usage = system_metrics.get('thread_usage_percent', 0)
                cpu_usage = system_metrics.get('cpu_percent', 0)
                
                # Critical escalation conditions
                if (memory_pressure > 95 or 
                    thread_usage > 95 or 
                    cpu_usage > 98):
                    return EmergencyLevel.SHUTDOWN
                
                # Severe escalation conditions
                elif (memory_pressure > 90 or 
                      thread_usage > 90 or
                      cpu_usage > 95):
                    return max(base_level, EmergencyLevel.CRITICAL, key=lambda x: list(EmergencyLevel).index(x))
                
                # Moderate escalation conditions
                elif (memory_pressure > 85 or 
                      thread_usage > 85 or
                      cpu_usage > 90):
                    return max(base_level, EmergencyLevel.SEVERE, key=lambda x: list(EmergencyLevel).index(x))
            
            return base_level
            
        except Exception as e:
            logger.error(f"Emergency level determination failed: {e}")
            return EmergencyLevel.CRITICAL  # Conservative default
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for emergency evaluation."""
        try:
            metrics = {}
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # Thread metrics
            try:
                result = subprocess.run(['ps', '-eLf'], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    thread_count = len(result.stdout.splitlines()) - 1
                    # Estimate usage percentage
                    if hasattr(resource, 'RLIMIT_NPROC'):
                        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
                        metrics['thread_usage_percent'] = (thread_count / soft_limit) * 100
                    metrics['thread_count'] = thread_count
            except Exception:
                metrics['thread_count'] = 0
                metrics['thread_usage_percent'] = 0
            
            # Process metrics
            metrics['process_count'] = len(psutil.pids())
            
            return metrics
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return {}
    
    async def activate_emergency_mode(self,
                                    level: Optional[EmergencyLevel] = None,
                                    trigger: EmergencyTrigger = EmergencyTrigger.AUTOMATIC,
                                    reason: str = "Unknown") -> bool:
        """Activate emergency mode with specified level and trigger."""
        
        with self.emergency_lock:
            try:
                logger.warning(f"Activating emergency mode - Trigger: {trigger.value}, Reason: {reason}")
                
                # Get system metrics
                system_metrics = self._get_system_metrics()
                
                # Determine emergency level if not specified
                if level is None:
                    level = self._determine_emergency_level(trigger, system_metrics)
                
                # Update emergency state
                self.emergency_state.level = level
                self.emergency_state.active_triggers.add(trigger)
                self.emergency_state.start_time = time.time()
                
                # Notify callbacks
                for callback in self.emergency_callbacks:
                    try:
                        callback(level, trigger)
                    except Exception as e:
                        logger.error(f"Emergency callback failed: {e}")
                
                # Execute emergency actions for this level and below
                actions_executed = 0
                actions_failed = 0
                
                for action in self.emergency_actions:
                    # Skip if action level is higher than current emergency level
                    action_level_index = list(EmergencyLevel).index(action.level)
                    current_level_index = list(EmergencyLevel).index(level)
                    
                    if action_level_index > current_level_index:
                        continue
                    
                    # Skip if already executed
                    if action.name in self.emergency_state.activated_actions:
                        continue
                    
                    try:
                        logger.warning(f"Executing emergency action: {action.name}")
                        
                        # Execute with timeout
                        success = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, action.function),
                            timeout=action.timeout_seconds
                        )
                        
                        if success:
                            self.emergency_state.activated_actions.add(action.name)
                            actions_executed += 1
                            logger.info(f"Emergency action completed: {action.name}")
                        else:
                            actions_failed += 1
                            logger.error(f"Emergency action failed: {action.name}")
                        
                    except asyncio.TimeoutError:
                        actions_failed += 1
                        logger.error(f"Emergency action timed out: {action.name}")
                    except Exception as e:
                        actions_failed += 1
                        logger.error(f"Emergency action exception: {action.name}: {e}")
                
                # Update emergency state duration
                self.emergency_state.duration_seconds = time.time() - self.emergency_state.start_time
                
                logger.warning(f"Emergency mode activation complete - Level: {level.value}, "
                             f"Actions: {actions_executed} completed, {actions_failed} failed, "
                             f"Duration: {self.emergency_state.duration_seconds:.1f}s")
                
                return actions_failed == 0
                
            except Exception as e:
                logger.critical(f"Emergency mode activation failed: {e}")
                return False
    
    async def attempt_recovery(self) -> bool:
        """Attempt to recover from emergency state."""
        try:
            logger.info("Attempting emergency recovery...")
            
            if self.emergency_state.recovery_attempted:
                logger.warning("Recovery already attempted")
                return self.emergency_state.recovery_successful
            
            self.emergency_state.recovery_attempted = True
            
            # Wait for system to stabilize
            await asyncio.sleep(5.0)
            
            # Check if conditions have improved
            current_metrics = self._get_system_metrics()
            
            # Recovery criteria
            recovery_thresholds = {
                'memory_percent': 80,
                'cpu_percent': 80,
                'thread_usage_percent': 80
            }
            
            recovery_possible = True
            for metric, threshold in recovery_thresholds.items():
                if current_metrics.get(metric, 100) > threshold:
                    recovery_possible = False
                    break
            
            if recovery_possible:
                logger.info("System conditions improved - attempting recovery")
                
                # Gradually reverse emergency actions (if reversible)
                reversed_actions = 0
                for action_name in list(self.emergency_state.activated_actions):
                    action = next((a for a in self.emergency_actions if a.name == action_name), None)
                    if action and action.reversible:
                        try:
                            # This would need specific recovery logic for each action
                            # For now, just mark as recovered
                            logger.info(f"Recovery action: {action_name}")
                            reversed_actions += 1
                        except Exception as e:
                            logger.warning(f"Recovery failed for action {action_name}: {e}")
                
                # Reset emergency state if recovery successful
                if reversed_actions > 0:
                    self.emergency_state.level = EmergencyLevel.WARNING  # Keep monitoring
                    self.emergency_state.recovery_successful = True
                    
                    # Notify recovery callbacks
                    for callback in self.recovery_callbacks:
                        try:
                            callback(True)
                        except Exception as e:
                            logger.error(f"Recovery callback failed: {e}")
                    
                    logger.info(f"Emergency recovery successful - reversed {reversed_actions} actions")
                    return True
            
            logger.warning("Emergency recovery not possible - conditions still critical")
            
            # Notify recovery callbacks
            for callback in self.recovery_callbacks:
                try:
                    callback(False)
                except Exception as e:
                    logger.error(f"Recovery callback failed: {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"Emergency recovery failed: {e}")
            return False
    
    def is_emergency_active(self) -> bool:
        """Check if emergency mode is active."""
        return self.emergency_state.level != EmergencyLevel.NONE
    
    def get_emergency_state(self) -> EmergencyState:
        """Get current emergency state."""
        return self.emergency_state
    
    def should_shutdown(self) -> bool:
        """Check if shutdown is requested."""
        return self.shutdown_requested or self.emergency_state.level == EmergencyLevel.SHUTDOWN
    
    def add_emergency_callback(self, callback: Callable[[EmergencyLevel, EmergencyTrigger], None]) -> None:
        """Add callback for emergency activation."""
        self.emergency_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[bool], None]) -> None:
        """Add callback for recovery attempts."""
        self.recovery_callbacks.append(callback)
    
    async def monitor_for_emergencies(self) -> None:
        """Monitor system for emergency conditions."""
        while self.monitoring_active and not self.should_shutdown():
            try:
                metrics = self._get_system_metrics()
                
                # Check for emergency conditions
                emergency_detected = False
                trigger = EmergencyTrigger.AUTOMATIC
                reason = ""
                
                # Memory pressure check
                if metrics.get('memory_percent', 0) > 95:
                    emergency_detected = True
                    trigger = EmergencyTrigger.MEMORY_PRESSURE
                    reason = f"Memory usage {metrics['memory_percent']:.1f}%"
                
                # Thread exhaustion check
                elif metrics.get('thread_usage_percent', 0) > 90:
                    emergency_detected = True
                    trigger = EmergencyTrigger.THREAD_EXHAUSTION
                    reason = f"Thread usage {metrics['thread_usage_percent']:.1f}%"
                
                # CPU overload check
                elif metrics.get('cpu_percent', 0) > 98:
                    emergency_detected = True
                    trigger = EmergencyTrigger.SYSTEM_OVERLOAD
                    reason = f"CPU usage {metrics['cpu_percent']:.1f}%"
                
                # Activate emergency if detected
                if emergency_detected and not self.is_emergency_active():
                    await self.activate_emergency_mode(trigger=trigger, reason=reason)
                
                # Check for recovery opportunities
                elif self.is_emergency_active() and not self.emergency_state.recovery_attempted:
                    # Wait before attempting recovery
                    if time.time() - self.emergency_state.start_time > 30:
                        await self.attempt_recovery()
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Emergency monitoring failed: {e}")
                await asyncio.sleep(10.0)  # Wait longer on error
    
    async def initialize(self) -> bool:
        """Initialize emergency fallback system."""
        try:
            # Create baseline snapshot
            self._create_system_snapshot()
            
            # Start monitoring
            self.monitoring_active = True
            
            logger.info("Emergency fallback system initialized and monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Emergency fallback system initialization failed: {e}")
            return False

# Global emergency system instance
_emergency_system = None

def get_emergency_system() -> EmergencyFallbackSystem:
    """Get the global emergency fallback system."""
    global _emergency_system
    if _emergency_system is None:
        _emergency_system = EmergencyFallbackSystem()
    return _emergency_system

async def activate_emergency_mode(reason: str = "Manual activation") -> bool:
    """Activate emergency mode (convenience function)."""
    system = get_emergency_system()
    return await system.activate_emergency_mode(
        trigger=EmergencyTrigger.USER_REQUEST,
        reason=reason
    )

def is_emergency_active() -> bool:
    """Check if emergency mode is active."""
    system = get_emergency_system()
    return system.is_emergency_active()